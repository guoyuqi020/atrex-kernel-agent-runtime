"""Complete argument schema for the Runtime command-line interface."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atrex-kernel-agent-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve the trusted Runtime API")
    serve.add_argument("--config", required=True, help="trusted Runtime JSON config")
    bootstrap = commands.add_parser(
        "bootstrap",
        help="initialize a commit-anchored Campaign",
    )
    bootstrap.add_argument("--config", required=True, help="trusted Runtime JSON config")
    bootstrap.add_argument("--campaign", required=True, help="trusted Campaign JSON definition")
    seed_lineage = commands.add_parser(
        "seed-lineage",
        help="add a Lineage rooted at existing Agent and Kernel content",
    )
    seed_lineage.add_argument("--config", required=True, help="trusted Runtime JSON config")
    seed_lineage.add_argument("--campaign", required=True, help="existing active Campaign ID")
    seed_lineage.add_argument("--spec", required=True, help="trusted Lineage seed JSON spec")
    run_campaign = commands.add_parser(
        "run-campaign",
        help="create or resume configured DSL lineages through an epoch target",
    )
    run_campaign.add_argument("--config", required=True, help="trusted Runtime JSON config")
    target = run_campaign.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--lineage",
        action="append",
        help="lineage ID to schedule; repeat for parallel DSL lineages",
    )
    target.add_argument(
        "--campaign",
        help="Campaign ID whose registered lineages should be discovered",
    )
    run_campaign.add_argument(
        "--target-epoch",
        required=True,
        type=int,
        help="absolute epoch number to complete",
    )
    run_campaign.add_argument(
        "--finalize",
        action="store_true",
        help="mark the Campaign and all discovered lineages completed after the target",
    )
    dev_shell = commands.add_parser(
        "dev-shell",
        help="open zsh/bash in a real Optimizer workspace without starting an Agent",
    )
    dev_shell.add_argument("--config", required=True, help="trusted Runtime JSON config")
    dev_target = dev_shell.add_mutually_exclusive_group(required=True)
    dev_target.add_argument(
        "--lineage",
        help="create or reuse the first Active Attempt in the lineage's current Epoch",
    )
    dev_target.add_argument(
        "--attempt",
        help="create a fresh run workspace for an existing running Attempt",
    )
    dev_shell.add_argument(
        "--shell",
        choices=("zsh", "bash"),
        default="zsh",
        help="interactive shell executable family (default: zsh)",
    )
    temporary_dev_shell = commands.add_parser(
        "temporary-dev-shell",
        help="open a disposable Optimizer workspace without Bootstrap",
    )
    temporary_dev_shell.add_argument("--config", required=True, help="trusted Runtime JSON config")
    temporary_dev_shell.add_argument(
        "--campaign", required=True, help="trusted single-DSL Campaign input definition"
    )
    temporary_dev_shell.add_argument(
        "--shell",
        choices=("zsh", "bash"),
        default="zsh",
        help="interactive shell executable family (default: zsh)",
    )
    temporary_evolver_dev_shell = commands.add_parser(
        "temporary-evolver-dev-shell",
        help="open a disposable Evolver workspace without Bootstrap",
    )
    temporary_evolver_dev_shell.add_argument(
        "--config", required=True, help="trusted Runtime JSON config"
    )
    temporary_evolver_dev_shell.add_argument(
        "--campaign", required=True, help="trusted single-DSL Campaign input definition"
    )
    temporary_evolver_dev_shell.add_argument(
        "--shell",
        choices=("zsh", "bash"),
        default="zsh",
        help="interactive shell executable family (default: zsh)",
    )
    evolver_dev_shell = commands.add_parser(
        "evolver-dev-shell",
        help="open zsh/bash in a frozen Evolver workspace for an existing Epoch",
    )
    evolver_dev_shell.add_argument("--config", required=True, help="trusted Runtime JSON config")
    evolver_dev_shell.add_argument("--lineage", required=True, help="existing Lineage ID")
    evolver_dev_shell.add_argument(
        "--epoch",
        required=True,
        type=int,
        help="existing absolute Epoch number whose Evolution input should be reconstructed",
    )
    evolver_dev_shell.add_argument(
        "--shell",
        choices=("zsh", "bash"),
        default="zsh",
        help="interactive shell executable family (default: zsh)",
    )
    cancel_campaign = commands.add_parser(
        "cancel-campaign",
        help="cancel a Campaign whose lineages are all quiescent",
    )
    cancel_campaign.add_argument("--config", required=True, help="trusted Runtime JSON config")
    cancel_campaign.add_argument("--campaign", required=True, help="Campaign ID to cancel")
    run_task_worker = commands.add_parser(
        "run-task-worker",
        help="claim and execute durable Campaign tasks",
    )
    run_task_worker.add_argument("--config", required=True, help="trusted Runtime JSON config")
    run_task_worker.add_argument(
        "--watch",
        action="store_true",
        help="continue polling instead of processing at most one task",
    )
    recover_epoch = commands.add_parser(
        "recover-epoch",
        help="authorize an idempotent retry of one failed epoch",
    )
    recover_epoch.add_argument("--config", required=True, help="trusted Runtime JSON config")
    recover_epoch.add_argument("--epoch", required=True, help="failed Epoch ID")
    recover_epoch.add_argument(
        "--recovery-key",
        required=True,
        help="operator-supplied idempotency key",
    )
    recover_epoch.add_argument(
        "--reason",
        required=True,
        help="operator justification recorded in the audit log",
    )
    list_epochs = commands.add_parser(
        "list-epochs",
        help="list every Active-versus-Challenger competition and winner",
    )
    list_epochs.add_argument("--config", required=True, help="trusted Runtime JSON config")
    epoch_scope = list_epochs.add_mutually_exclusive_group(required=True)
    epoch_scope.add_argument("--campaign", help="Campaign ID to enumerate")
    epoch_scope.add_argument("--lineage", help="Lineage ID to enumerate")
    list_epochs.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
        help="machine-readable JSON or a human-readable competition history (default: json)",
    )
    list_attempts = commands.add_parser(
        "list-attempts",
        help="list every optimization Attempt, including no-Candidate outcomes",
    )
    list_attempts.add_argument("--config", required=True, help="trusted Runtime JSON config")
    attempt_scope = list_attempts.add_mutually_exclusive_group(required=True)
    attempt_scope.add_argument("--campaign", help="Campaign ID to enumerate")
    attempt_scope.add_argument("--lineage", help="Lineage ID to enumerate")
    list_attempts.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
        help="machine-readable JSON or a human-readable Attempt history (default: json)",
    )
    show_attempt = commands.add_parser(
        "show-attempt",
        help="show one durable Attempt and its terminal disposition",
    )
    show_attempt.add_argument("--config", required=True, help="trusted Runtime JSON config")
    show_attempt.add_argument("--attempt", required=True, help="Attempt ID")
    list_sessions = commands.add_parser(
        "list-worker-sessions",
        help="list model-backed Worker processes and their retained raw traces",
    )
    list_sessions.add_argument("--config", required=True, help="trusted Runtime JSON config")
    session_scope = list_sessions.add_mutually_exclusive_group(required=True)
    session_scope.add_argument("--campaign", help="Campaign ID to enumerate")
    session_scope.add_argument("--lineage", help="Lineage ID to enumerate")
    session_scope.add_argument("--epoch", help="Epoch ID to enumerate")
    session_scope.add_argument("--attempt", help="Attempt or Bootstrap Attempt ID")
    session_scope.add_argument("--subject", help="Generalization or other Worker subject ID")
    list_sessions.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
    )
    show_session = commands.add_parser(
        "show-worker-session",
        help="show one Worker lifecycle record and raw session-trace digest",
    )
    show_session.add_argument("--config", required=True, help="trusted Runtime JSON config")
    show_session.add_argument("--session", required=True, help="Worker session ID")
    list_kernels = commands.add_parser(
        "list-kernels",
        help="list every terminal Kernel and authoritative evaluation",
    )
    list_kernels.add_argument("--config", required=True, help="trusted Runtime JSON config")
    kernel_scope = list_kernels.add_mutually_exclusive_group(required=True)
    kernel_scope.add_argument("--campaign", help="Campaign ID to enumerate")
    kernel_scope.add_argument("--lineage", help="Lineage ID to enumerate")
    list_kernels.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
        help="machine-readable JSON or a human-readable version history (default: json)",
    )
    show_kernel = commands.add_parser(
        "show-kernel",
        help="show one Kernel, its producing Agent, and all durable measurements",
    )
    show_kernel.add_argument("--config", required=True, help="trusted Runtime JSON config")
    show_kernel.add_argument("--kernel", required=True, help="Kernel revision ID")
    list_agents = commands.add_parser(
        "list-agent-revisions",
        help="list every lineage-local Kernel Agent revision",
    )
    list_agents.add_argument("--config", required=True, help="trusted Runtime JSON config")
    agent_scope = list_agents.add_mutually_exclusive_group(required=True)
    agent_scope.add_argument("--campaign", help="Campaign ID to enumerate")
    agent_scope.add_argument("--lineage", help="Lineage ID to enumerate")
    list_agents.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "table"),
        default="json",
        help="machine-readable JSON or a human-readable version history (default: json)",
    )
    show_agent = commands.add_parser(
        "show-agent-revision",
        help="show one versioned Kernel Agent revision",
    )
    show_agent.add_argument("--config", required=True, help="trusted Runtime JSON config")
    show_agent.add_argument(
        "--agent-revision",
        required=True,
        help="Kernel Agent revision ID",
    )
    list_bootstrap_runs = commands.add_parser(
        "list-bootstrap-runs",
        help="list every retained execution generation for one Bootstrap Attempt",
    )
    list_bootstrap_runs.add_argument("--config", required=True)
    list_bootstrap_runs.add_argument("--attempt", required=True, help="Bootstrap Attempt ID")
    show_bootstrap_run = commands.add_parser(
        "show-bootstrap-run",
        help="show one exact Bootstrap execution generation",
    )
    show_bootstrap_run.add_argument("--config", required=True)
    show_bootstrap_run.add_argument("--attempt", required=True, help="Bootstrap Attempt ID")
    show_bootstrap_run.add_argument("--generation", required=True, type=int)
    list_evaluations = commands.add_parser(
        "list-evaluations",
        help="list every Agent and Runtime-final evaluation for one Attempt",
    )
    list_evaluations.add_argument("--config", required=True)
    list_evaluations.add_argument("--attempt", required=True)
    show_evaluation = commands.add_parser(
        "show-evaluation",
        help="show one immutable evaluated Kernel/outcome pair",
    )
    show_evaluation.add_argument("--config", required=True)
    show_evaluation.add_argument("--evaluation", required=True)
    show_evaluation.add_argument("--source", action="store_true")
    show_evaluation.add_argument("--result", action="store_true")
    list_trials = commands.add_parser(
        "list-kernel-trials",
        help="list every exact experimental Kernel snapshot observed in one Attempt",
    )
    list_trials.add_argument("--config", required=True)
    list_trials.add_argument("--attempt", required=True)
    show_trial = commands.add_parser(
        "show-kernel-trial",
        help="show one Kernel Trial and optionally its exact source files",
    )
    show_trial.add_argument("--config", required=True)
    show_trial.add_argument("--trial", required=True)
    show_trial.add_argument("--source", action="store_true")
    show_trial.add_argument("--result", action="store_true")
    artifact_gc = commands.add_parser(
        "gc-artifacts",
        help="inspect or collect unreferenced Artifacts while the Runtime is stopped",
    )
    artifact_gc.add_argument("--config", required=True)
    artifact_gc.add_argument("--minimum-age-seconds", required=True, type=float)
    artifact_gc.add_argument("--limit", required=True, type=int)
    artifact_gc.add_argument(
        "--apply",
        action="store_true",
        help="delete eligible objects; the default is a dry run",
    )
    artifact_gc.add_argument(
        "--confirm-runtime-stopped",
        action="store_true",
        help="confirm all Runtime, Worker, Wiki drainer, and bootstrap processes are stopped",
    )
    workspace_gc = commands.add_parser(
        "gc-workspaces",
        help="inspect or collect old Worker run directories while the Runtime is stopped",
    )
    workspace_gc.add_argument("--config", required=True)
    workspace_gc.add_argument("--minimum-age-seconds", required=True, type=float)
    workspace_gc.add_argument("--limit", required=True, type=int)
    workspace_gc.add_argument("--apply", action="store_true", help="delete eligible runs")
    workspace_gc.add_argument(
        "--confirm-runtime-stopped",
        action="store_true",
        help="confirm all Runtime, Worker, Wiki drainer, and bootstrap processes are stopped",
    )
    digest_evolver = commands.add_parser(
        "digest-evolver-bundle",
        help="validate an Evolver Bundle and print its canonical content SHA-256",
    )
    digest_evolver.add_argument("--path", required=True, help="Evolver Bundle root")
    digest_evolver.add_argument("--max-files", type=int, default=1024)
    digest_evolver.add_argument("--max-bytes", type=int, default=8388608)
    return parser
