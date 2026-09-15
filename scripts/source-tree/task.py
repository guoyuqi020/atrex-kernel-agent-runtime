#!/usr/bin/env python3
"""Prepare or run an offline source-tree task kit without changing its inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.composition.bootstrap import build_optimizer_base_loader
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.abba import CommitPinnedAtrexBenchEvaluator
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.gateway.production_policy import ProductionKernelPolicy
from atrex_runtime.kernel_sources import SourceManifest, import_source_tree, source_path
from atrex_runtime.workers.evolver_bundle import GitEvolverBundleResolver
from atrex_runtime.workers.problem_generalization import validate_public_operator_contract

REPOSITORY = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clone_bundle(bundle: Path, destination: Path, expected_commit: str) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", str(bundle), str(destination)], check=True, capture_output=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected_commit:
        raise ValueError(f"{bundle.name} resolved to {actual}, expected {expected_commit}")


def verify_assets(root: Path, expected: dict[str, str]) -> None:
    """Check task-owned hashes recorded against the supplied original packages."""
    for name, digest in expected.items():
        path = root / source_path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Task asset is missing or not a regular file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Task asset differs from the verified original: {name}")


def verify_component(root: Path, expected: dict[str, str]) -> None:
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD"], text=True
    ).splitlines()
    if set(tracked) != set(expected):
        raise ValueError(f"Packaged {root.name} file set differs from the verified original")
    verify_assets(root, expected)


def prepare(inputs: Path, workspace: Path, backend: str | None, port: int | None) -> None:
    if workspace.exists():
        raise ValueError("Workspace already exists; resume it or choose a new --workspace")
    if (
        workspace.is_relative_to(REPOSITORY / "data")
        or workspace.is_relative_to(inputs)
        or inputs.is_relative_to(workspace)
    ):
        raise ValueError("Workspace must be separate from task inputs; use workspaces/FA4")
    integrity_path = inputs / "asset-integrity.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.is_file() else None
    if integrity is not None:
        verify_assets(inputs, integrity["files_sha256"])
    definition = json.loads((inputs / "campaign.json").read_text())
    CampaignSpecV3.model_validate(definition)
    if len(definition["lineages"]) != 1:
        raise ValueError("This source-tree task runner requires exactly one DSL Lineage")
    manifest = SourceManifest.model_validate_json(
        (inputs / "task/source_manifest.json").read_bytes()
    )
    template = json.loads((inputs / "runtime.template.json").read_text())
    campaign = template["campaign"]
    if port is not None:
        template["server"]["port"] = port
        campaign["gateway_proxy_url"] = f"http://127.0.0.1:{port}"
    template["agate"]["base_url"] = os.environ.get("AGATE_URL") or template["agate"]["base_url"]
    for role in ("optimizer", "evolver"):
        if backend is not None:
            campaign[role]["agent_backend"] = backend
        if sys.platform == "linux":
            # The sandbox hides Home, so use the global interpreter behind the active venv.
            campaign[role]["command_prefix"] = [str(Path(sys.executable).resolve())]
    for section in (template["kernel_agent"]["base_source"], campaign["evolver"]):
        section["repository"] = "./" + os.path.relpath(
            (inputs / section["repository"]).resolve(), workspace
        )
    reference_root = campaign["launcher"]["container"]["reference_projects_root"]
    campaign["launcher"]["container"]["reference_projects_root"] = "./" + os.path.relpath(
        (inputs / reference_root).resolve(), workspace
    )
    # Credentials are copied into isolated per-Session homes by the normal Runtime launcher.
    campaign["launcher"]["backend_credentials"]["host_home"] = None

    workspace.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".source-tree-prepare-", dir=workspace.parent
    ) as temporary:
        stage = Path(temporary) / "workspace"
        stage.mkdir()
        if integrity is not None:
            shutil.copyfile(integrity_path, stage / "asset-integrity.json")
        for directory in ("task", "initial-evidence"):
            source = inputs / directory
            if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
                raise ValueError(f"Task inputs cannot contain symlinks: {directory}")
            shutil.copytree(
                source, stage / directory, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__")
            )
        if (inputs / "smoke").is_dir():
            shutil.copytree(inputs / "smoke", stage / "smoke")
        clone_bundle(inputs / "source.bundle", stage / "source", manifest.source.revision)
        clone_bundle(
            inputs / "evaluator.bundle",
            stage / "evaluator",
            campaign["gate_policy"]["evaluator"]["commit"],
        )
        if integrity is not None:
            for name in ("source", "evaluator"):
                component = integrity["components"][name]
                actual = subprocess.check_output(
                    ["git", "-C", str(stage / name), "rev-parse", "HEAD"], text=True
                ).strip()
                if actual != component["commit"]:
                    raise ValueError(f"Packaged {name} Commit differs from verified original")
                verify_component(stage / name, component["files_sha256"])
        gate = campaign["gate_policy"]
        task = stage / "task"
        shapes = json.loads((task / "shape_valid.json").read_text())
        public = json.loads((task / "shape_train.json").read_text())
        validate_public_operator_contract(public, private_shapes=shapes)
        if (stage / "smoke/reference/shapes.json").is_file():
            smoke_shapes = json.loads((stage / "smoke/reference/shapes.json").read_text())
            normalized = {
                str(row["id"]): {key: value for key, value in row.items() if key != "id"}
                for row in smoke_shapes["shapes"]
            }
            if normalized != shapes:
                raise ValueError("Smoke Shape contract differs from full evaluation Shapes")
        contract = AgateEvaluationContractV1(
            candidate_path="kernel.py",
            reference_py=(task / "reference.py").read_text(),
            input_py=(task / "input.py").read_text(),
            shapes=shapes,
            metadata=json.loads((task / "metadata.json").read_text()),
            roofline=json.loads((task / "roofline.json").read_text()),
            options={
                "num_correctness_cases": gate["retention"]["correctness_cases"],
                "bench_iters": gate["retention"]["bench_iters"],
                "atol": gate["atol"],
                "rtol": gate["rtol"],
                "timeout_s": gate["evaluation_timeout_seconds"],
            },
            lock_clocks=gate["lock_clocks"],
        )
        write_json(stage / "campaign.json", definition)
        write_json(stage / "evaluation-contract.json", contract.model_dump(mode="json"))
        write_json(stage / "runtime.json", template)
        # Validate paths against their final directory while loading the staged local evaluator.
        final_settings = RuntimeSettings.model_validate(template)
        assert final_settings.campaign is not None
        final_settings = final_settings.model_copy(
            update={
                "kernel_agent": final_settings.kernel_agent.resolve_from(workspace),
                "campaign": final_settings.campaign.resolve_from(workspace),
            }
        )
        CampaignSpecV3.from_file(stage / "campaign.json")
        with tempfile.TemporaryDirectory(prefix="atrex-source-tree-preflight-") as check_dir:
            artifacts = LocalArtifactStore(Path(check_dir) / "artifacts")
            imported = import_source_tree(
                task / "source_manifest.json", stage / "source", artifacts
            )
            source = artifacts.verify(imported.seed_digest)
            file_count = len(imported.validate_tree(source.payload_path))
            policy_violations = ProductionKernelPolicy().violations(
                source.payload_path, "kernel.py", Dsl(next(iter(definition["lineages"]))), imported
            )
            if gate["production_gate"] and policy_violations:
                raise ValueError("Production gate rejected seed: " + "; ".join(policy_violations))
            loader = build_optimizer_base_loader(final_settings, artifacts)
            if loader is None:
                raise ValueError("Optimizer base source is required")
            optimizer = loader.build_candidate(
                Dsl(next(iter(definition["lineages"]))), definition["base_revision"]["commit"]
            )
            evolver = final_settings.campaign.evolver
            resolved_evolver = GitEvolverBundleResolver(
                artifacts,
                repository=evolver.repository,
                commit=evolver.commit,
                git_executable=evolver.git_executable,
                fetch_timeout_seconds=evolver.fetch_timeout_seconds,
                max_archive_bytes=evolver.max_archive_bytes,
                command_prefix=evolver.command_prefix,
                max_files=evolver.max_bundle_files,
                max_bytes=evolver.max_bundle_bytes,
            ).resolve()
            # This consumer needs the staged checkout, not the unpublished final workspace.
            evaluator = CommitPinnedAtrexBenchEvaluator(
                **{**gate["evaluator"], "repository": str(stage / "evaluator")}
            )
            evaluator_files = evaluator.files()
            write_json(
                stage / "prepared.json",
                {
                    "operator": definition["operator"],
                    "hardware_target": definition["hardware_target"],
                    "dsl": next(iter(definition["lineages"])),
                    "private_shape_count": len(shapes),
                    "source_commit": manifest.source.revision,
                    "source_artifact_digest": str(imported.seed_digest),
                    "source_file_count": file_count,
                    "optimizer_commit": definition["base_revision"]["commit"],
                    "optimizer_artifact_digest": str(optimizer.candidate.optimizer_digest),
                    "evolver_commit": resolved_evolver.commit,
                    "evaluator_commit": evaluator.commit,
                    "evaluator_file_count": len(evaluator_files),
                    "evaluator_bundle_digest": evaluator.bundle_digest(),
                    "input_files_sha256": {
                        path.relative_to(inputs).as_posix(): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in sorted(inputs.rglob("*"))
                        if path.is_file() and path.name != ".DS_Store"
                    },
                    "production_gate": gate["production_gate"],
                    "seed_static_policy_violations": list(policy_violations),
                    "gpu_jobs_submitted": 0,
                },
            )
        stage.rename(workspace)
    RuntimeSettings.from_file(workspace / "runtime.json")
    print(f"Prepared: {workspace}\nSource, Optimizer, Evolver and evaluator preflight passed.")
    print(
        f"Target: {definition['hardware_target']} / {next(iter(definition['lineages']))}; "
        f"{len(shapes)} private Shapes. R0 still requires Bootstrap bring-up."
    )
    print("No services, model sessions or GPU jobs were started.")


def run_smoke(workspace: Path, mode: str, shape_id: str) -> None:
    """Run the unchanged supplied Agate smoke scripts, not a Runtime acceptance gate."""
    smoke = workspace / "smoke"
    integrity = json.loads((workspace / "asset-integrity.json").read_text())
    verify_assets(
        workspace,
        {
            name: digest
            for name, digest in integrity["files_sha256"].items()
            if name.startswith("smoke/")
            or name in {"task/adapter.py", "task/reference.py", "task/input.py"}
        },
    )
    shapes = json.loads((smoke / "reference/shapes.json").read_text())
    if mode == "target" and shape_id not in {str(row["id"]) for row in shapes["shapes"]}:
        raise ValueError(f"Unknown smoke Shape ID: {shape_id}; expected 0..29")
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=workspace) as temporary:
        root = Path(temporary)
        shutil.copytree(smoke, root, dirs_exist_ok=True)
        original_source = integrity["components"]["source"]["files_sha256"]
        verify_assets(workspace / "source", original_source)
        for name in original_source:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace / "source" / name, destination)
        shutil.copyfile(workspace / "task/adapter.py", root / "kernel.py")
        for name in ("reference.py", "input.py"):
            shutil.copyfile(workspace / "task" / name, root / "reference" / name)
        subprocess.run(["bash", str(root / "run_agate_dev.sh"), mode, shape_id], check=True)


def service_secrets(workspace: Path) -> dict[str, str]:
    path = workspace / "runtime-secrets.json"
    with (workspace / ".runtime-secrets.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".runtime-secrets-", dir=workspace)
            try:
                with os.fdopen(descriptor, "w") as output:
                    json.dump(
                        {
                            "ATREX_CAPABILITY_SIGNING_KEY": secrets.token_urlsafe(48),
                            "ATREX_ADMIN_BEARER_TOKEN": secrets.token_hex(32),
                        },
                        output,
                    )
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        value = json.loads(path.read_text())
        if (
            not isinstance(value, dict)
            or set(value) != {"ATREX_CAPABILITY_SIGNING_KEY", "ATREX_ADMIN_BEARER_TOKEN"}
            or any(not isinstance(item, str) or not item for item in value.values())
        ):
            raise ValueError("Invalid saved Runtime secrets; refusing to regenerate")
        return value


def run_cli(arguments: list[str], *, output: Path | None = None) -> object:
    command = [sys.executable, "-c", "from atrex_runtime.cli import main; main()", *arguments]
    if output is None:
        subprocess.run(command, check=True)
        return None
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    value = json.loads(result.stdout)
    write_json(output, value)
    print(result.stdout, flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "role", choices=("prepare", "smoke", "serve", "bootstrap", "campaign", "inspect")
    )
    parser.add_argument("--inputs", type=Path, default=REPOSITORY / "data/FA4")
    parser.add_argument("--workspace", type=Path, default=REPOSITORY / "workspaces/FA4")
    parser.add_argument("--backend", choices=("claude", "codex", "qodercli", "pi"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--target-epoch", type=int, default=1)
    parser.add_argument(
        "--smoke-mode", choices=("upstream-p128", "target"), default="upstream-p128"
    )
    parser.add_argument("--shape-id", default="0")
    args = parser.parse_args()
    if args.target_epoch < 1 or (args.port is not None and not 1 <= args.port <= 65535):
        parser.error("target-epoch must be positive and port must be 1..65535")
    if args.role != "prepare" and (args.backend is not None or args.port is not None):
        parser.error(
            "backend and port are frozen during prepare; use a new workspace to change them"
        )
    workspace = args.workspace.resolve()
    if args.role == "prepare":
        prepare(args.inputs.resolve(), workspace, args.backend, args.port)
        return
    config = workspace / "runtime.json"
    if not config.is_file():
        parser.error("Prepare this workspace first")
    RuntimeSettings.from_file(config)
    if args.role == "smoke":
        run_smoke(workspace, args.smoke_mode, args.shape_id)
        return
    os.environ.update(service_secrets(workspace))
    if args.role == "serve":
        from atrex_runtime.cli import main as runtime_main

        runtime_main(["serve", "--config", str(config)])
        return
    if args.role in {"bootstrap", "campaign"}:
        result = run_cli(
            ["bootstrap", "--config", str(config), "--campaign", str(workspace / "campaign.json")],
            output=workspace / "bootstrap-result.json",
        )
        if args.role == "bootstrap":
            return
        run_cli(
            [
                "run-campaign",
                "--config",
                str(config),
                "--campaign",
                result["campaign_id"],
                "--target-epoch",
                str(args.target_epoch),
            ],
            output=workspace / "epoch-result.json",
        )
        return
    result = json.loads((workspace / "bootstrap-result.json").read_text())
    run_cli(
        [
            "list-kernels",
            "--config",
            str(config),
            "--campaign",
            result["campaign_id"],
            "--format",
            "table",
        ]
    )


if __name__ == "__main__":
    main()
