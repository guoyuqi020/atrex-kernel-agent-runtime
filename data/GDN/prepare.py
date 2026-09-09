#!/usr/bin/env python3
"""Prepare the local GDN kit in Linux/Lima; never start services, Agents or GPU jobs."""

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

from atrex_runtime.artifacts.local import LocalArtifactStore
from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.contract import AgateEvaluationContractV1
from atrex_runtime.gateway.production_policy import ProductionKernelPolicy
from atrex_runtime.kernel_sources import import_source_tree
from atrex_runtime.workers.problem_generalization import validate_public_operator_contract


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True, stderr=subprocess.PIPE).strip()


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() == text:
        return
    # Local launch inputs are regenerated only before Runtime state has been created.
    if path.exists() and (path.parent / "state").exists():
        raise SystemExit(f"Existing runtime state: refusing to replace {path.name}")
    path.write_text(text)


def main() -> None:
    if sys.platform != "linux":
        raise SystemExit("Run this preparation inside Lima Ubuntu, not with the macOS .venv.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qodercli", "claude", "codex", "pi"))
    parser.add_argument("--worker-user", help="existing non-root Linux user; defaults to this user")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    task = root / "task"
    campaign = CampaignSpecV3.from_file(root / "campaign.json")
    if campaign.hardware_target != "L20D":
        raise SystemExit("This prepared GDN campaign targets L20D.")
    template = json.loads((root / "runtime.template.json").read_text())
    worker = pwd.getpwnam(
        args.worker_user or os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    )
    if worker.pw_uid == 0:
        raise SystemExit("Choose an existing non-root --worker-user.")
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
    launcher["sandbox"]["worker_user"] = worker.pw_name
    launcher["backend_credentials"]["host_home"] = worker.pw_dir
    template["agate"]["base_url"] = os.environ.get("AGATE_URL") or template["agate"]["base_url"]
    config_path = root / "runtime.json"
    RuntimeSettings.model_validate(template)

    # This is an offline bundle of one initial commit, not a link to the external repro.
    manifest = json.loads((task / "source_manifest.json").read_text())
    revision = manifest["source"]["revision"]
    source = root / "source"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--quiet", str(root / "source.bundle"), str(source)],
            check=True,
            capture_output=True,
        )
    if git("-C", str(source), "rev-parse", "HEAD") != revision:
        raise SystemExit("Local source HEAD differs from the pinned GDN seed; not overwriting it.")
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

    write_json(config_path, template)
    write_json(root / "evaluation-contract.json", contract.model_dump(mode="json"))
    settings = RuntimeSettings.from_file(config_path)
    assert settings.campaign is not None and settings.kernel_agent.base_source is not None
    repositories = (
        (settings.kernel_agent.base_source.repository, campaign.base_revision.commit),
        (settings.campaign.evolver.repository, settings.campaign.evolver.commit),
        (
            settings.campaign.gate_policy.evaluator.repository,
            settings.campaign.gate_policy.evaluator.commit,
        ),
    )
    for repository, commit in repositories:
        git("-C", repository, "cat-file", "-e", f"{commit}^{{commit}}")
    provenance = {
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
    }
    write_json(root / "prepared.json", provenance)
    print(f"Prepared: {root}")
    print(f"GPU=L20D DSL=cutedsl backend={settings.campaign.optimizer.agent_backend}")
    print(f"Source: {revision}; {file_count} files; {len(shapes)} private shapes")
    print(f"Source integrity and production policy: passed ({digest})")
    print(f"Worker: {worker.pw_name}; sandbox=bwrap+cgroup; Python={worker_python}")
    for executable_name in ("bwrap", "systemd-run", settings.campaign.optimizer.agent_backend):
        print(f"{executable_name}: {shutil.which(executable_name) or 'NOT IN PATH'}")
    print("No services, Agent sessions or GPU jobs were started.")


if __name__ == "__main__":
    main()
