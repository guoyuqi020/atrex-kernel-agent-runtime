#!/usr/bin/env python3
"""Prepare a GDN workspace from read-only task inputs; never start services or Agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gdn_workspace import binding_for, service_config

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.composition.bootstrap import build_optimizer_base_loader, build_roofline_builder
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.abba import CommitPinnedAtrexBenchEvaluator
from atrex_runtime.gateway.configuration import agate_settings_from_environment
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.gateway.production_policy import ProductionKernelPolicy
from atrex_runtime.kernel_sources import import_source_tree
from atrex_runtime.workers.evolver_bundle import GitEvolverBundleResolver
from atrex_runtime.workers.problem_generalization import validate_public_operator_contract


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True, stderr=subprocess.PIPE).strip()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def write_inputs(workspace: Path, files: dict[str, bytes]) -> None:
    """Preflight the entire snapshot before writing; never change a registered run's inputs."""
    if any((workspace / name).exists() for name in (
        "state", "service-binding.json", "service.json",
    )):
        for name, content in files.items():
            path = workspace / name
            if not path.is_file() or path.read_bytes() != content:
                raise SystemExit(
                    f"Existing runtime state: refusing to replace {name}. "
                    "Resume with run.py, or prepare a new --workspace for changed inputs."
                )
    for name, content in files.items():
        path = workspace / name
        if path.is_file() and path.read_bytes() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def rebase_repositories(template: dict, inputs: Path, workspace: Path) -> None:
    """Keep repository references relative even with a custom workspace depth."""
    campaign = template["campaign"]
    for section, key in (
        (template["kernel_agent"]["base_source"], "repository"),
        (campaign["evolver"], "repository"),
        (campaign["gate_policy"]["evaluator"], "repository"),
        (campaign["roofline_builder"], "repository"),
    ):
        value = section[key]
        if value and (key != "repository" or value.startswith(("./", "../", "/"))):
            section[key] = "./" + os.path.relpath((inputs / value).resolve(), workspace)


def worker_for_mode(mode: str, requested: str | None, configured: str | None = None):
    """Container workers run as the invoking user; only sandbox mode switches users."""
    if mode == "container":
        worker = pwd.getpwuid(os.getuid())
        if requested is not None and requested != worker.pw_name:
            raise SystemExit(
                "Container mode uses the current user; --worker-user cannot switch users"
            )
        return worker
    if mode != "sandbox":
        raise SystemExit(f"Unsupported GDN launcher mode: {mode}")
    if configured is not None and requested is not None and requested != configured:
        raise SystemExit("--worker-user differs from the shared service configuration")
    worker = pwd.getpwnam(
        configured or requested or os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    )
    if worker.pw_uid == 0:
        raise SystemExit("Sandbox mode needs an existing non-root --worker-user")
    return worker


def validate_runtime_sources(
    settings: RuntimeSettings, campaigns: tuple[CampaignSpecV3, ...],
) -> dict:
    """Use the actual consumers, not cat-file, without creating Runtime state or running code."""
    campaign = settings.campaign
    if campaign is None:
        raise ValueError("GDN preparation requires Campaign runtime settings")
    checked: dict = {"optimizers": []}
    stage = "Optimizer"
    try:
        with tempfile.TemporaryDirectory(prefix="atrex-gdn-bundle-check-") as temporary:
            artifacts = LocalArtifactStore(Path(temporary) / "artifacts")
            loader = build_optimizer_base_loader(settings, artifacts)
            if loader is None:
                raise ValueError("Optimizer base source is required")
            revisions = {(dsl, spec.base_revision.commit)
                         for spec in campaigns for dsl in spec.lineages}
            for dsl, commit in sorted(revisions):
                stage = f"Optimizer {dsl.value} {commit}"
                imported = loader.build_candidate(dsl, commit)
                checked["optimizers"].append({
                    "dsl": dsl.value, "commit": commit,
                    "artifact_digest": str(imported.candidate.optimizer_digest),
                })
            evolver = campaign.evolver
            stage = f"Evolver {evolver.commit}"
            imported_evolver = GitEvolverBundleResolver(
                artifacts, repository=evolver.repository, commit=evolver.commit,
                git_executable=evolver.git_executable,
                fetch_timeout_seconds=evolver.fetch_timeout_seconds,
                max_archive_bytes=evolver.max_archive_bytes,
                command_prefix=evolver.command_prefix,
                max_files=evolver.max_bundle_files, max_bytes=evolver.max_bundle_bytes,
            ).resolve()
            checked["evolver"] = {
                "commit": imported_evolver.commit,
                "artifact_digest": str(imported_evolver.artifact_digest),
            }
            gate = settings.gate_policy or campaign.gate_policy
            stage = f"Evaluator {gate.evaluator.commit}"
            evaluator = CommitPinnedAtrexBenchEvaluator(
                **gate.evaluator.model_dump(exclude={"agate_package_version"})
            )
            files = evaluator.files()
            checked["evaluator"] = {
                "commit": gate.evaluator.commit,
                "bundle_digest": evaluator.bundle_digest(), "file_count": len(files),
            }
            if campaign.roofline_builder is not None:
                stage = f"Roofline {campaign.roofline_builder.commit}"
                roofline = build_roofline_builder(settings)
                assert roofline is not None
                roofline.validate_source()
                checked["roofline"] = {"commit": campaign.roofline_builder.commit}
    except Exception as error:
        raise ValueError(f"Runtime source preflight failed ({stage}): {error}") from error
    return checked


def main() -> None:
    if sys.platform != "linux":
        raise SystemExit("Run this preparation inside Lima Ubuntu, not with the macOS .venv.")
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--inputs", type=Path, default=repository / "data/GDN",
        help="task input directory (default: data/GDN)",
    )
    parser.add_argument(
        "--workspace", type=Path,
        help="run output directory (default: workspaces/<input directory name>)",
    )
    parser.add_argument("--backend", choices=("qodercli", "claude", "codex", "pi"))
    parser.add_argument("--worker-user", help="sandbox Worker; container must match current user")
    parser.add_argument("--services-only", action="store_true", help="prepare a shared Runtime")
    parser.add_argument("--service-workspace", type=Path, help="attach to a prepared service")
    parser.add_argument("--port", type=int, help="Runtime port; service/standalone preparation")
    args = parser.parse_args()
    if args.services_only and args.service_workspace:
        parser.error("--services-only and --service-workspace are mutually exclusive")
    if args.port is not None and (args.service_workspace or not 1 <= args.port <= 65535):
        parser.error("--port must be 1..65535 and cannot override an attached service")
    root = args.inputs.resolve()
    workspace = (args.workspace or repository / "workspaces" / (
        "control-gdn" if args.services_only else root.name
    )).resolve()
    if workspace.is_relative_to(repository / "data") or root.is_relative_to(workspace):
        raise SystemExit("--workspace must be separate from data; use workspaces/GDN.")
    task = root / "task"
    campaign = CampaignSpecV3.from_file(root / "campaign.json")
    if campaign.hardware_target != "L20D":
        raise SystemExit("This prepared GDN campaign targets L20D.")
    binding = None
    if args.service_workspace:
        if any((workspace / name).exists() for name in (
            "runtime.json", "state", "runtime-secrets.json",
        )):
            raise SystemExit("Cannot attach a standalone workspace; prepare a new task workspace")
        binding = binding_for(workspace, args.service_workspace.resolve())
        shared_config = service_config(args.service_workspace.resolve())
        template = json.loads(shared_config.read_text())
        settings = RuntimeSettings.from_file(shared_config)
        assert settings.campaign is not None
        if args.backend and any(
            section.agent_backend != args.backend
            for section in (settings.campaign.optimizer, settings.campaign.evolver)
        ):
            raise SystemExit("--backend differs from the shared service configuration")
        launcher = settings.campaign.launcher
        worker = worker_for_mode(
            launcher.mode, args.worker_user,
            None if launcher.sandbox is None else launcher.sandbox.worker_user,
        )
        executable = str(settings.campaign.roofline_builder.python_executable)
        worker_python = settings.campaign.optimizer.command_prefix[0]
    else:
        template, worker, executable, worker_python = configure_runtime(root, workspace, args)
        settings = RuntimeSettings.model_validate(template)
        assert settings.campaign is not None
        settings = settings.model_copy(update={
            "kernel_agent": settings.kernel_agent.resolve_from(workspace),
            "campaign": settings.campaign.resolve_from(workspace),
        })
    if args.services_only:
        if any((workspace / name).exists() for name in ("campaign.json", "service-binding.json")):
            raise SystemExit("Cannot turn a task workspace into a shared service")
        checked = validate_runtime_sources(
            settings, (campaign, CampaignSpecV3.from_file(root / "ablation-campaign.json")),
        )
        config_bytes = json_bytes(template)
        write_inputs(workspace, {
            "runtime.json": config_bytes,
            "service.json": json_bytes({
                "kind": "gdn-shared-runtime",
                "runtime_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "source_preflight": checked,
            }),
        })
        print(f"Shared Runtime workspace: {workspace}")
        print(f"Runtime endpoint: {settings.campaign.gateway_proxy_url}")
        print(f"Worker launcher: {settings.campaign.launcher.mode}")
        print("Prepared only; no task inputs, services, Agent sessions or GPU jobs were created.")
        return

    # This is an offline bundle of one initial commit, not a link to the external repro.
    manifest = json.loads((task / "source_manifest.json").read_text())
    revision = manifest["source"]["revision"]
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "source"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--quiet", str(root / "source.bundle"), str(source)],
            check=True,
            capture_output=True,
        )
    if git("-C", str(source), "rev-parse", "HEAD") != revision:
        raise SystemExit(
            "Workspace seed differs from the current task input; not overwriting it. "
            "Resume with run.py, or prepare a new --workspace for changed inputs."
        )
    if git("-C", str(source), "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit(
            "Local source seed is dirty; do not edit it. Agents edit work/kernel instead."
        )

    gate = template["campaign"]["gate_policy"]
    shapes = json.loads((task / "shape_valid.json").read_text())
    public = json.loads((task / "shape_train.json").read_text())
    validate_public_operator_contract(public, private_shapes=shapes)
    contract = AgateEvaluationContractV1(
        candidate_path="kernel.py",
        reference_py=(task / "reference.py").read_text(),
        input_py=(task / "input.py").read_text(),
        shapes=shapes,
        metadata=json.loads((task / "metadata.json").read_text()),
        roofline=json.loads((task / "roofline.json").read_text()),
        options={
            "num_correctness_cases": 5,
            "bench_iters": gate["bootstrap"]["bench_iters"],
            "atol": gate["atol"],
            "rtol": gate["rtol"],
            "timeout_s": gate["evaluation_timeout_seconds"],
        },
        lock_clocks=gate["lock_clocks"],
    )
    # Validate locks and production policy with no persistent Runtime state and no GPU imports.
    with tempfile.TemporaryDirectory(prefix="atrex-gdn-input-check-") as temporary:
        artifacts = LocalArtifactStore(Path(temporary) / "artifacts")
        source_contract = import_source_tree(task / "source_manifest.json", source, artifacts)
        artifact = artifacts.verify(source_contract.seed_digest)
        ProductionKernelPolicy().validate(
            artifact.payload_path, "kernel.py", Dsl.CUTEDSL, source_contract
        )
        file_count = len(source_contract.validate_tree(artifact.payload_path))
        digest = str(source_contract.seed_digest)

    assert settings.campaign is not None and settings.kernel_agent.base_source is not None
    source_preflight = validate_runtime_sources(
        settings, (campaign, CampaignSpecV3.from_file(root / "ablation-campaign.json")),
    )
    provenance = {
        "task_inputs": os.path.relpath(root, workspace),
        "hardware_target": campaign.hardware_target,
        "dsl": "cutedsl",
        "source_commit": revision,
        "source_artifact_digest": digest,
        "source_file_count": file_count,
        "private_shape_count": len(shapes),
        "source_bundle_sha256": hashlib.sha256((root / "source.bundle").read_bytes()).hexdigest(),
        "task_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(task.iterdir())
            if path.is_file()
        },
        "optimizer_commit": campaign.base_revision.commit,
        "evolver_commit": settings.campaign.evolver.commit,
        "evaluator_commit": settings.campaign.gate_policy.evaluator.commit,
        "agent_backend": settings.campaign.optimizer.agent_backend,
        "worker_user": worker.pw_name,
        "python": executable,
        "worker_python": worker_python,
        "prepared_on": "Linux/Lima",
        "gpu_jobs_submitted": 0,
        "source_preflight": source_preflight,
    }
    files = {
        "evaluation-contract.json": json_bytes(contract.model_dump(mode="json")),
        "prepared.json": json_bytes(provenance),
    }
    if binding is None:
        files["runtime.json"] = json_bytes(template)
    else:
        files["service-binding.json"] = json_bytes(binding)
    for name in ("campaign.json", "ablation-campaign.json", "ablation.json"):
        files[name] = (root / name).read_bytes()
    for directory in ("task", "initial-evidence"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"Task input must not be a symlink: {path}")
            if path.is_file() and path.name != ".DS_Store" and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    write_inputs(workspace, files)
    print(f"Task inputs: {root}")
    print(f"Prepared workspace: {workspace}")
    if binding is not None:
        print(f"Shared Runtime config: {shared_config}")
    print(f"GPU=L20D DSL=cutedsl backend={settings.campaign.optimizer.agent_backend}")
    print(f"Source: {revision}; {file_count} files; {len(shapes)} private shapes")
    print(f"Source integrity and production policy: passed ({digest})")
    print("Runtime Bundle preflight: Optimizer, Evolver, Evaluator and Roofline passed")
    mode = settings.campaign.launcher.mode
    print(f"Worker: {worker.pw_name}; launcher={mode}; Python={worker_python}")
    executables = ["bwrap", settings.campaign.optimizer.agent_backend]
    if mode == "sandbox":
        executables.append("systemd-run")
    for executable_name in executables:
        print(f"{executable_name}: {shutil.which(executable_name) or 'NOT IN PATH'}")
    print("No services, Agent sessions or GPU jobs were started.")


def configure_runtime(root: Path, workspace: Path, args: argparse.Namespace) -> tuple:
    """Resolve a deployment template once; attached tasks never override it."""
    template = json.loads((root / "runtime.template.json").read_text())
    worker = worker_for_mode(template["campaign"]["launcher"]["mode"], args.worker_user)
    executable = str(Path(sys.executable).absolute())  # Runtime-side dependencies use the venv.
    # Worker entrypoints use only the standard library. Resolve the venv symlink
    # to the global interpreter, since Sandbox masks the host's /home tree.
    worker_python = str(Path(sys.executable).resolve())
    for name in ("optimizer", "evolver"):
        template["campaign"][name]["command_prefix"] = [worker_python]
        if args.backend:
            template["campaign"][name]["agent_backend"] = args.backend
    template["campaign"]["roofline_builder"]["python_executable"] = executable
    launcher = template["campaign"]["launcher"]
    if launcher["mode"] == "sandbox":
        launcher["sandbox"]["worker_user"] = worker.pw_name
    launcher["backend_credentials"]["host_home"] = worker.pw_dir
    template["agate"] = agate_settings_from_environment(
        os.environ, base_url=os.environ.get("AGATE_URL") or template["agate"]["base_url"]
    ).model_dump(exclude_none=True)
    if args.port is not None:
        template["server"]["port"] = args.port
        template["campaign"]["gateway_proxy_url"] = f"http://127.0.0.1:{args.port}"
    rebase_repositories(template, root, workspace)
    RuntimeSettings.model_validate(template)

    return template, worker, executable, worker_python


if __name__ == "__main__":
    main()
