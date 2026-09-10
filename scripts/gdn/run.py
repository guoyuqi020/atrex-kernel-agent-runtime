#!/usr/bin/env python3
"""Run the prepared GDN Runtime or Bootstrap plus a bounded Campaign in Linux."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from gdn_workspace import load_service_secrets, resolve_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument("role", choices=("serve", "campaign", "ablation"))
    parser.add_argument("--workspace", type=Path, default=repository / "workspaces/GDN")
    parser.add_argument("--service-workspace", type=Path, help="verify the saved service binding")
    parser.add_argument("--target-epoch", type=int, default=100)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Run inside Lima Ubuntu with the Linux venv.")
    if args.target_epoch < 1:
        raise SystemExit("--target-epoch must be positive")
    root = args.workspace.resolve()
    if root.is_relative_to(repository / "data") or repository.is_relative_to(root):
        raise SystemExit("--workspace must be separate from data; use workspaces/GDN.")
    service, config = resolve_service(root, args.service_workspace)
    if args.role == "serve" and service != root:
        raise SystemExit(f"Start the shared Runtime once with serve --workspace {service}")
    if args.role != "serve" and (root / "service.json").is_file():
        raise SystemExit("Select a task workspace for campaign/ablation, not the service workspace")
    os.environ.update(load_service_secrets(service))
    print(f"Runtime config: {config}", flush=True)
    cli = Path(sys.executable).absolute().parent / "atrex-kernel-agent-runtime"
    if args.role == "serve":
        os.execv(cli, [str(cli), "serve", "--config", str(config)])
    if args.role == "ablation":
        runner = repository / "scripts/source-tree/run.py"
        os.execv(sys.executable, [
            sys.executable, str(runner), "--config", str(config),
            "--campaign", str(root / "ablation-campaign.json"),
            "--plan", str(root / "ablation.json"), "--workspace", str(root / "ablation"),
            "--target-epoch", str(args.target_epoch),
        ])
    # Bootstrap is idempotent: a resumed run reuses its sealed Campaign and v0.
    print("Bootstrap: L20D / CuteDSL", flush=True)
    completed = subprocess.run(
        [str(cli), "bootstrap", "--config", str(config), "--campaign", str(root / "campaign.json")],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    result = json.loads(completed.stdout)
    (root / "bootstrap-result.json").write_text(json.dumps(result, indent=2) + "\n")
    campaign = result["campaign_id"]
    print(f"Campaign: {campaign}; running through Epoch {args.target_epoch}", flush=True)
    completed = subprocess.run(
        [
            str(cli),
            "run-campaign",
            "--config",
            str(config),
            "--campaign",
            campaign,
            "--target-epoch",
            str(args.target_epoch),
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    (root / "epoch-result.json").write_text(completed.stdout)
    print(completed.stdout, flush=True)


if __name__ == "__main__":
    main()
